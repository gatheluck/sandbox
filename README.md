# Sandbox

[![CI](https://github.com/gatheluck/sandbox/workflows/CI/badge.svg)](https://github.com/gatheluck/sandbox/actions?query=workflow%3ACI)
[![MIT License](https://img.shields.io/github/license/gatheluck/sandbox?color=green)](LICENSE)

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