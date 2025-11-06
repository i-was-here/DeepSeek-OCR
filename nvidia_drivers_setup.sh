#!/bin/bash

sudo apt update
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices
sudo apt install -y nvidia-driver-570  # or recommended one
sudo reboot

