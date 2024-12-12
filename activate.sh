#!/bin/bash
source mri_env/bin/activate
export $(cat .env | xargs) 