#!/bin/bash
## install python 3.5.x
sudo yum install -y python35u python35u-pip python35u-libs python35u-devel python35u-pip python35u-setuptools

## install python packages
sudo pip3.5 install -r ./requirements.txt