import json


file_address = "/home/zsong/working/output.json"
with open(file_address) as infile:
    o = json.load(infile):q


'''
file_address = "/home/zsong/working/post_df.json"
with open(file_address) as infile:
  o = json.load(infile)
  for k in o['t3_1bx6qw'].keys():
    print(k)
'''
  '''
  chunkSize = 1
  with open('chunked_10' + '.json', 'w') as outfile:
      json.dump(o[: chunkSize], outfile)
  '''
