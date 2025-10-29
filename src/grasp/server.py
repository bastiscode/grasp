import asyncio
import json
import os
import random
import string
import threading
import time
from datetime import datetime, timezone
from logging import INFO, FileHandler, Logger
from typing import Any

from fastapi import WebSocketDisconnect
from pydantic import BaseModel, Field, conlist
from universal_ml_utils.io import dump_json, load_json
from universal_ml_utils.logging import get_logger
from universal_ml_utils.ops import partition_by

from grasp.configs import ServerConfig
from grasp.core import generate, load_notes, setup
from grasp.manager import find_embedding_model
from grasp.model import Message
from grasp.tasks import Task
from grasp.tasks.examples import load_example_indices

# keep track of connections and limit to 10 concurrent connections
active_connections = 0


class Past(BaseModel):
    messages: conlist(Message, min_length=1)  # type: ignore
    known: set[str]


class Request(BaseModel):
    task: Task
    input: Any
    knowledge_graphs: conlist(str, min_length=1)  # type: ignore
    past: Past | None = None


class State(BaseModel):
    task: Task
    selected_kgs: conlist(str, min_length=1) = Field(alias="selectedKgs")  # type: ignore
    last_input: Any | None = Field(default=None, alias="lastInput")
    last_output: dict[str, Any] | None = Field(default=None, alias="lastOutput")


ALPHABET = string.ascii_letters + string.digits


def generate_id(length: int = 6) -> str:
    chars = random.sample(ALPHABET, length)
    return f"grasp-{''.join(chars)}"


def serve(config: ServerConfig, log_level: int | str | None = None) -> None:
    # create a fast api websocket server to serve the generate_sparql function
    import uvicorn
    from fastapi import FastAPI, HTTPException, WebSocket

    app = FastAPI()
    logger = get_logger("GRASP SERVER", log_level)
    if config.log_outputs is not None:
        os.makedirs(os.path.dirname(config.log_outputs), exist_ok=True)
        output_logger = Logger("GRASP JSONL OUTPUTS")
        output_logger.addHandler(
            FileHandler(config.log_outputs, mode="a", encoding="utf-8")
        )
        output_logger.setLevel(INFO)
    else:
        output_logger = None

    # add cors
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    managers = setup(config)
    kgs = [manager.kg for manager in managers]

    model = find_embedding_model(managers)

    notes = {}
    kg_notes = {}
    example_indices = {}
    for task in Task:
        general_notes, kg_specific_notes = load_notes(config)
        notes[task.value] = general_notes
        kg_notes[task.value] = kg_specific_notes

        task_indices = load_example_indices(task.value, config, model=model)
        example_indices[task.value] = task_indices

    if config.share is not None:
        share_dir = config.share

        def generate_and_check_share_id(max_retries: int = 3) -> str | None:
            share_id = generate_id()
            if not os.path.exists(os.path.join(share_dir, f"{share_id}.json")):
                return share_id
            if max_retries <= 0:
                return None
            return generate_and_check_share_id(max_retries - 1)

        @app.post("/save")
        async def _save(request: State):
            share_id = generate_and_check_share_id()
            if share_id is None:
                logger.error(
                    "Failed to generate unique share ID after multiple attempts"
                )
                raise HTTPException(status_code=500, detail="Failed to share state")

            try:
                data = request.model_dump(by_alias=True)
                timestamp = datetime.now(timezone.utc).isoformat()
                dump_json(
                    {
                        "timestamp": timestamp,
                        "state": data,
                    },
                    os.path.join(share_dir, f"{share_id}.json"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to persist state: {exc}")
                raise HTTPException(
                    status_code=500, detail="Failed to save state"
                ) from exc

            logger.info(f"Saved state {share_id}")
            return {"id": share_id, "url": f"/load/{share_id}"}

        @app.get("/load/{share_id}")
        async def _load(share_id: str):
            try:
                path = os.path.join(share_dir, f"{share_id}.json")
                data = load_json(path)
                return State(**data["state"])
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="State not found")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Unexpected error loading state {share_id}: {exc}")
                raise HTTPException(status_code=500, detail="Failed to load state")

    @app.get("/knowledge_graphs")
    async def _knowledge_graphs():
        return kgs

    @app.get("/config")
    async def _config():
        return config.model_dump()

    @app.post("/run")
    async def _run(request: Request):
        global active_connections

        if active_connections >= config.max_connections:
            logger.warning(
                "HTTP run request refused: "
                f"maximum of {config.max_connections:,} active connections reached"
            )
            raise HTTPException(
                status_code=503,
                detail="Server too busy, try again later",
            )

        active_connections += 1
        logger.info(f"HTTP run request started ({active_connections=:,})")

        try:
            sel = request.knowledge_graphs
            if not sel or not all(kg in kgs for kg in sel):
                logger.error(
                    "Unsupported knowledge graph selection:\n"
                    f"{request.model_dump_json(indent=2)}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported knowledge graph selection",
                )

            sel_managers, _ = partition_by(managers, lambda m: m.kg in sel)

            past_messages = request.past.messages if request.past else None
            past_known = request.past.known if request.past else None

            def run_generate() -> dict:
                try:
                    *_, output = generate(
                        request.task,
                        request.input,
                        config,
                        sel_managers,
                        kg_notes[request.task],
                        notes[request.task],
                        example_indices[request.task],
                        past_messages,
                        past_known,
                        logger,
                    )
                except ValueError as exc:
                    raise RuntimeError("No output produced") from exc

                return output

            try:
                output = await asyncio.wait_for(
                    asyncio.to_thread(run_generate),
                    timeout=config.max_generation_time,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Generation hit time limit of {config.max_generation_time:,} seconds"
                )
                raise HTTPException(
                    status_code=504,
                    detail=(
                        f"Generation hit time limit of {config.max_generation_time:,} seconds"
                    ),
                )
            except HTTPException as e:
                raise e
            except Exception as exc:
                logger.error(f"Unexpected error with HTTP run request:\n{exc}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to handle request:\n{exc}",
                )

            if output_logger is not None:
                output_logger.info(json.dumps(output))

            return output

        finally:
            active_connections -= 1
            logger.info(f"HTTP run request finished ({active_connections=:,})")

    @app.websocket("/live")
    async def _live(websocket: WebSocket):
        global active_connections
        assert websocket.client is not None
        client = f"{websocket.client.host}:{websocket.client.port}"
        await websocket.accept()

        # Check if we've reached the maximum number of connections
        if active_connections >= config.max_connections:
            logger.warning(
                f"Connection from {client} immediately closed: "
                f"maximum of {config.max_connections:,} active connections reached"
            )
            await websocket.close(code=1013, reason="Server too busy, try again later")
            return

        active_connections += 1
        logger.info(f"{client} connected ({active_connections=:,})")
        last_active = time.perf_counter()

        async def idle_checker():
            nonlocal last_active
            while True:
                await asyncio.sleep(min(5, config.max_idle_time))

                if time.perf_counter() - last_active <= config.max_idle_time:
                    continue

                msg = f"Connection closed due to inactivity after {config.max_idle_time:,} seconds"
                logger.info(f"{client}: {msg}")
                await websocket.close(code=1013, reason=msg)  # Try Again Later
                break

        idle_task = asyncio.create_task(idle_checker())

        try:
            while True:
                data = await websocket.receive_json()
                last_active = time.perf_counter()
                try:
                    request = Request(**data)
                except Exception:
                    logger.error(
                        f"Invalid request from {client}:\n{json.dumps(data, indent=2)}"
                    )
                    await websocket.send_json({"error": "Invalid request format"})
                    continue

                sel = request.knowledge_graphs
                if not sel or not all(kg in kgs for kg in sel):
                    logger.error(
                        f"Unsupported knowledge graph selection by {client}:\n"
                        f"{request.model_dump_json(indent=2)}"
                    )
                    await websocket.send_json(
                        {"error": "Unsupported knowledge graph selection"}
                    )
                    continue

                logger.info(
                    f"Processing request from {client}:\n"
                    f"{request.model_dump_json(indent=2)}"
                )

                sel_managers, _ = partition_by(managers, lambda m: m.kg in sel)

                past_messages = None
                past_known = None
                if request.past is not None:
                    # set past
                    past_messages = request.past.messages
                    past_known = request.past.known

                loop = asyncio.get_running_loop()
                queue = asyncio.Queue()
                stop_event = threading.Event()

                def run_generate() -> None:
                    try:
                        generator = generate(
                            request.task,
                            request.input,
                            config,
                            sel_managers,
                            kg_notes[request.task],
                            notes[request.task],
                            example_indices[request.task],
                            past_messages,
                            past_known,
                            logger,
                        )

                        for output in generator:
                            if stop_event.is_set():
                                break
                            asyncio.run_coroutine_threadsafe(
                                queue.put(("data", output)),
                                loop,
                            ).result()

                    except Exception as exc:
                        asyncio.run_coroutine_threadsafe(
                            queue.put(("error", exc)),
                            loop,
                        ).result()
                    finally:
                        asyncio.run_coroutine_threadsafe(
                            queue.put(("done", None)),
                            loop,
                        ).result()

                producer = asyncio.create_task(asyncio.to_thread(run_generate))
                #
                # Track start time for timeout
                start_time = time.perf_counter()

                try:
                    while True:
                        kind, payload = await queue.get()

                        if kind == "data":
                            # Check if we've exceeded the time limit
                            current_time = time.perf_counter()
                            if current_time - start_time > config.max_generation_time:
                                msg = f"Generation hit time limit of {config.max_generation_time:,} seconds"
                                logger.warning(msg)
                                stop_event.set()
                                await websocket.send_json({"error": msg})
                                break

                            output = payload
                            if output["type"] == "output" and output_logger is not None:
                                output_logger.info(json.dumps(output))

                            await websocket.send_json(output)
                            data = await websocket.receive_json()
                            last_active = time.perf_counter()

                            if data.get("cancel", False):
                                logger.info(f"Generation cancelled by {client}")
                                stop_event.set()
                                await websocket.send_json({"cancelled": True})
                                break

                        elif kind == "error":
                            exc = payload
                            stop_event.set()
                            logger.error(
                                f"Unexpected error while generating for {client}:\n{exc}"
                            )
                            await websocket.send_json(
                                {"error": f"Failed to handle request:\n{exc}"}
                            )
                            break

                        elif kind == "done":
                            break

                finally:
                    stop_event.set()
                    try:
                        await producer
                    except Exception as exc:
                        logger.error(f"Generator worker for {client} failed:\n{exc}")

        except WebSocketDisconnect:
            pass

        except Exception as e:
            logger.error(f"Unexpected error with {client}:\n{e}")
            await websocket.send_json({"error": f"Failed to handle request:\n{e}"})

        finally:
            idle_task.cancel()
            active_connections -= 1
            logger.info(f"{client} disconnected ({active_connections=:,})")

    uvicorn.run(app, host="0.0.0.0", port=config.port)
