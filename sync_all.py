#!/usr/bin/env python3
import json
import argparse
from tap import Connector

# args
parser = argparse.ArgumentParser()
parser.add_argument("--sync", action="store_true", help="run sync phase")
parser.add_argument("--warmup", action="store_true", help="run warm-up phase")
parser.add_argument("--include-data", action="store_true", help="include 'data' in codebase")
args = parser.parse_args()

SYNC_CODE = lambda client, codebase: [client.sync_code(b) for b in codebase]
names = Connector().list_all()
clients = [Connector(n) for n in names]

## Default sync
with open('manifest.json') as f:
    manifest = json.load(f)
    codebase = list(manifest['codebase'].keys())

## remove data in codebase (unless requested)
if not args.include_data:
    codebase = [b for b in codebase if b != 'data']

# (keep your original flow—names/clients re-listed)
names = Connector().list_all()
clients = [Connector(n) for n in names]
default_code_base = ['manifest']

if args.sync:
    for c in clients:
        print(c.client)
        SYNC_CODE(c, default_code_base)
        c.reload()
        SYNC_CODE(c, codebase)

## warm up
if args.warmup:
    conn = Connector()
    clients = conn.list_all()
    conns = [Connector(c) for c in clients]
    for c in conns:
        conn.batch(c.client, 'warm_up', {})
    conn.executor.fetch()
    while True:
        try:
            outputs = conn.executor.apply()
            break
        except:
            continue
    print(outputs)
