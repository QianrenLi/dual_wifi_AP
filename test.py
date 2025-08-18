from util.exp_setup import create_transmission_config
from tap import Connector

conn = Connector()
# for client in conn.list_all():
#     Connector(client).sync_file_pull("stream-replay/logs/test.txt")

create_transmission_config("system_verify", conn)
