# Sandbox

## Directory configuration

### /applications
Applications are placed here. `/applications/sandbox` is an example application. If you want to add new application, please refere that.

### /provision
Scripts for environment set up are placed here.

## How to use

### Running Docker Compose
- Move to `/provision/environments/development`.

- Command `sudo docker-compose up` will start Docker Compose.

### Open terminal in sandbox container
- Move to `/provision/environments/development`.

- Command `sudo docker-compose exec sandbox bash` open a new terminal in sandbox container.