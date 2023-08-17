import json
import os
import boto3

lamb = boto3.client('lambda')

edl_token = os.environ['EDL_TOKEN']
scenes = [
    'S1B_IW_SLC__1SDV_20201115T162313_20201115T162340_024278_02E29D_5C54',
    'S1A_IW_SLC__1SDV_20201203T162353_20201203T162420_035524_042744_6D5C',
]

function_arn = 'ffwilliams2-test-indexer-IndexCreationFunction-qTfV0eMha9PT'

results = []
for scene in scenes:
    payload = {'scene': scene, 'edl_token': edl_token}
    result = lamb.invoke(FunctionName=function_arn, InvocationType='Event', Payload=json.dumps(payload))
    if result['StatusCode'] != 202:
        print(f'Invocation failed with for {scene} with status code {result["StatusCode"]}')
    else:
        print(f'{scene} successfully submitted!')
