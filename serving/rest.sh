#!/bin/bash
# An example REST call
echo '{"body": "the cloud", "subject" : "is malfunctioning"}' | curl -H "Content-Type: application/json" -d @- http://localhost:5000/predict
