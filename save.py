import requests
from contextlib import closing
import json
import os

data_dir = "data"
data_log = "data.log"
os.makedirs(data_dir, exist_ok=True)


def action(base, act, params={}):
    r = requests.get('/'.join((base, 'api/3/action', act)), params=params)
    return r

def get_group(base, group):
    remaining = None
    per = 10
    page = 0
    f_log = open(data_log,'w')
    while remaining is None or remaining > 0:
        params = {'rows': min(per, (remaining or per)), 'start': page * per}
        if group:
            params['q'] = 'groups:' + group

        resp = action(base, 'package_search', params=params)
        try:
            resp = resp.json()
        except:
            f_log.write("error: page " + str(page) + " " + str(params) + '\n')
            continue

        if remaining is None:
            remaining = resp['result']['count']
            print("total datasets: ", remaining)

        print("page:", page)

        remaining -= per
        page += 1

        for result in resp['result']['results']:
            # remove resources from result so we can just look at the dataset info on its own
            resources = result['resources']
            del result['resources']

            parent = False

            path = os.path.join(data_dir,result['id'])

            for resource in resources:
                if resource['format'] == 'CSV':# : and resource['url'].endswith('.csv'):
                    print("processing: " + resource['url'])
                    if not parent:
                        parent = True
                        if not os.path.exists(path):
                            os.makedirs(path)
                        if not os.path.exists(os.path.join(path, "meta.json")):
                            with open(os.path.join(path, "meta.json"), "w") as ofile:
                                json.dump(result, ofile)

                    fpath = os.path.join(path, resource['id'])
                    if not os.path.exists(fpath):
                        os.makedirs(fpath, exist_ok=True)
                    if not os.path.exists(os.path.join(fpath, "meta.json")):
                        with open(os.path.join(fpath, "meta.json"), "w") as ofile:
                            json.dump(resource, ofile)

                    csv = os.path.join(fpath, "data.csv")
                    if not os.path.isfile(csv) and not os.path.exists(csv):
                        try:
                            with open(csv, "wb") as ofile:
                                    with closing(requests.get(resource['url'], stream=True, verify=False)) as request:
                                        if request.ok:
                                            for block in request.iter_content(1024):
                                                ofile.write(block)
                        except Exception as e:
                            print(resource['url'], e)
                            f_log.write("error: page " + str(page) + " " + str(params) + ' ' + resource['url']+'\n')
                            os.remove(csv)

# just download as much of everything as we care to get
get_group('http://catalog.data.gov', None)
