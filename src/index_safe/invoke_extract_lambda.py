import json
import os
import boto3

lamb = boto3.client('lambda')

edl_token = os.environ['EDL_TOKEN']
bursts = [
    'S1B_IW_SLC__1SDV_20201115T162313_20201115T162340_024278_02E29D_5C54_IW3_VV_4',
]

function_arn = (
    'arn:aws:lambda:us-west-2:050846374571:function:ffwilliams2-test-indexer-BurstExtractFunction-kcaujwt92zHk'
)

results = []
for burst in bursts:
    payload = {'burst': burst, 'edl_token': edl_token}
    result = lamb.invoke(FunctionName=function_arn, InvocationType='Event', Payload=json.dumps(payload))
    if result['StatusCode'] != 202:
        print(f'Invocation failed with for {burst} with status code {result["StatusCode"]}')
    else:
        print(f'{burst} successfully submitted!')
