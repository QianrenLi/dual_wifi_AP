#!/usr/bin/env python3
import socket
from typing import Dict
import json
from util.control_cmd import ControlCmd


class ipc_socket():
    def __init__(self, ip_addr, ipc_port, local_port=12345, link_name=""):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 196)
        self.sock.settimeout(1.5)
        self.sock.bind(("0.0.0.0", local_port))
        self.link_name = link_name
        self.ip_addr = ip_addr
        self.ipc_port = ipc_port

    def send_cmd(self, *args):
        cmd = args[0]
        body = args[1]
        message = {"cmd": {cmd: body}}
        server_address = (self.ip_addr, self.ipc_port)
        message = json.dumps(message)
        self.sock.sendto(message.encode(), server_address)
    
    def ipc_communicate(self, *args):
        self.send_cmd(*args)
        _buffer, addr = self.sock.recvfrom(2048)
        return _buffer
    
    def ipc_transmit(self, *args):
        self.send_cmd(*args)
        
    def close(self):
        self.sock.close()

class ipc_control(ipc_socket):
    def __init__(self, ip_addr, ipc_port, local_port=12345, link_name=""):
        super().__init__(ip_addr, ipc_port, local_port, link_name)
    
    def throttle(self, throttle_ctl):
        self.ipc_transmit('Throttle', throttle_ctl)
        return None

    def statistics(self):
        res = self.ipc_communicate('Statistics', {})
        return json.loads(res)["cmd"]["Statistics"]
    
    def policy_parameters(self, policy_parameters):
        self.ipc_transmit('PolicyParameters', policy_parameters)
        return None
    
    def control(self, control_cmd: Dict[str, ControlCmd]):
        control_cmd_json = {key: dict(value) for key, value in control_cmd.items()}
        self.ipc_transmit('Control', control_cmd_json)
        return None
    
    def release(self):
        self.close()
        return None
    
if __name__ == '__main__':
    test_ipc = ipc_control( '127.0.0.1',  11112 )
    res = test_ipc.statistics()
    print(res)
    # from util.control_cmd import list_to_cmd, cmd_to_list
    # # test_ipc.control(
    # #     {'6203@128': 
    # #         ControlCmd(
    # #             policy_parameters=[0.1, 0.2, 0.3, 0.5],
    # #             version=1
    # #         )
    # #     }
    # # )
    
    # test = list_to_cmd(ControlCmd, [0.1, 0.2, 0.3, 0.5, 1])
    # print(test)
    # print(cmd_to_list(test))
    # test_ipc.control(
    #     {'6203@128': 
    #         list_to_cmd(ControlCmd, [0.1, 0.2, 0.3, 0.5, 2])
    #     }
    # )
    
    # res = test_ipc.statistics()
    
    # test_ipc.throttle({'6203@128': 0.5})