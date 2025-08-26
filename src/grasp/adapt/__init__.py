import os
import random

import yaml
from tqdm import tqdm, trange
from universal_ml_utils.io import dump_json, load_jsonl
from universal_ml_utils.logging import get_logger
from universal_ml_utils.ops import partition_by

from grasp.adapt.note_taking import take_notes
from grasp.adapt.utils import link
from grasp.configs import Adapt
from grasp.core import generate, setup
from grasp.utils import Sample


def adapt(
    task: str,
    config: Adapt,
    out_dir: str,
    overwrite: bool = False,
    log_level: str | int | None = None,
) -> None:
    if os.path.exists(out_dir) and not overwrite:
        raise FileExistsError(f"Output directory {out_dir} already exists")

    assert config.method == "iterative_note_taking", (
        "Only iterative_note_taking method is supported for adaptation"
    )

    logger = get_logger("GRASP ADAPTATION", log_level)

    managers, notes = setup(config)

    assert config.seed is not None, "Seed must be set for adaptation"

    inputs: list[tuple[str, Sample]] = []
    for ipt in config.input:
        samples = [(ipt.kg, Sample(**sample)) for sample in load_jsonl(ipt.file)]
        if config.samples_per_file is not None:
            random.seed(config.seed)
            random.shuffle(samples)
            samples = samples[: config.samples_per_file]
        inputs.extend(samples)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(config.model_dump(), f)

    out_file = os.path.join(out_dir, "notes.general.json")

    for r in trange(config.num_rounds, desc="Adapting GRASP to KGs"):
        random.seed(config.seed + r)
        if inputs:
            samples = random.sample(inputs, min(config.samples_per_round, len(inputs)))
        else:
            raise NotImplementedError(
                "Adaptation without input samples is not implemented yet"
            )

        outputs = []
        for kg, sample in tqdm(samples, desc=f"Round {r + 1} samples", leave=False):
            sel_managers, _ = partition_by(managers, lambda m: m.kg == kg)
            assert len(sel_managers) == 1, (
                f"Expected exactly one manager for kg {kg}, got {len(sel_managers)}"
            )

            *_, output = generate(task, sample.question, config, sel_managers, notes)
            outputs.append(output)

        take_notes(samples, outputs, managers, notes, config, logger)

        for manager in managers:
            out_file = os.path.join(out_dir, f"notes.{manager.kg}.round_{r}.json")
            dump_json(manager.notes, out_file, indent=2)
            link(out_file, os.path.join(out_dir, f"notes.{manager.kg}.json"))

        out_file = os.path.join(out_dir, f"notes.general.round_{r}.json")
        dump_json(notes, out_file, indent=2)
        link(out_file, os.path.join(out_dir, "notes.general.json"))
