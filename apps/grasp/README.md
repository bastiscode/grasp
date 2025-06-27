# GRASP web app

Web app for the GRASP server.

## Setup

1. Change the config options in [the config file](lib/config.dart)
to match your GRASP server setup.

2. Build the docker image:
`docker build -t grasp-app .`

3. Run the docker container:
`docker run --name grasp-app -d --restart unless-stopped -p <port>:80 grasp-app`

You can also build and run the app locally without Docker:

```bash
# Make sure you have Flutter installed
flutter build web --release

# Start a local server
python -m http.server --directory build/web <port>
```
