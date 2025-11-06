#!/bin/bash

sudo apt-get remove --purge 'nvidia-*'
sudo apt-get autoremove
sudo apt-get autoclean

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices
sudo apt install -y nvidia-driver-570  # or recommended one
sudo reboot

