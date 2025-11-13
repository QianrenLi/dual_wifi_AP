from ipaddress import ip_network

import subprocess as sp
import psutil
import shutil
import socket
import time
import getpass

SHELL_POPEN = lambda x: sp.Popen(x, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
SHELL_RUN = lambda x: sp.run(x, stdout=sp.PIPE, stderr=sp.PIPE, check=True, shell=True)




_IP = shutil.which("ip") or "/usr/sbin/ip"

class SudoIP:
    def __init__(self, password: str):
        # keep in a mutable buffer so we can overwrite later
        self._pw = bytearray(password.encode("utf-8"))

    def _pw_bytes(self):
        # sudo expects the password followed by newline
        return bytes(self._pw) + b"\n"

    def wipe(self):
        # overwrite the buffer to reduce the chance of lingering in memory
        for i in range(len(self._pw)):
            self._pw[i] = 0

    def run(self, *args, check=True):
        """Run: sudo ip <args...> with password via stdin."""
        # -S  : read password from stdin
        # -p "" : empty prompt (keeps stderr clean)
        # NOTE: no shell=True to avoid leaks/quoting issues
        result = sp.run(
            ["sudo", "-S", "-p", "", _IP] + list(args),
            input=self._pw_bytes(),
            stdout=sp.PIPE,
            stderr=sp.PIPE,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"sudo ip {' '.join(args)} failed: {result.stderr.decode().strip()}"
            )
        return result

    def try_run(self, *args):
        """Run and ignore failures (useful for cleanup deletes)."""
        return self.run(*args, check=False)


def seperate_nic(sudoip:SudoIP, winf_names, netmask=24):
    initial_priority = 100

    # Get all network interface addresses
    while True:
        net_addrs = psutil.net_if_addrs()
        nic_info = {}
        for nic in winf_names:
            nic = nic.strip()
            if not nic:
                continue
            
            addresses = net_addrs[nic]
            for snicaddr in addresses:
                if snicaddr.family == socket.AF_INET:
                    ip_addr = ip_network(f"{snicaddr.address}/{netmask}", strict=False)
                    nic_info[nic] = {
                        'ip': snicaddr.address,
                        'netmask': snicaddr.netmask,
                        'subnet': str(ip_addr),
                        'gateway': str(ip_addr[1]),
                        'priority': initial_priority
                    }
            if nic_info.get(nic) is None:
                # If no IPv4 address found, retry
                initial_priority = 100
                print(f"Retrying for {nic} as no IPv4 address found.")
                time.sleep(1)
                break
            initial_priority += 1
        if len(nic_info) == len(winf_names):
            break

    for nic, info in nic_info.items():
            # sudoip.try_run(cmd)
        sudoip.try_run("route", "del", "default", "via", info["gateway"], "dev", nic, "src", info["ip"])
        # sudoip.try_run("route", "del", info["subnet"], "dev", nic, "table", str(info["priority"]))
        # sudoip.try_run("route", "del", "default", "via", info["gateway"], "dev", nic, "table", str(info["priority"]))
        sudoip.try_run("route", "del", info["subnet"], "dev", nic)
        sudoip.try_run("rule",  "del", "from", info["ip"], "lookup", str(info["priority"]))
        sudoip.try_run("rule",  "del", "table", str(info["priority"]))
        sudoip.try_run("route", "flush", "table", str(info["priority"]))
        
        
        sudoip.run("rule",  "add", "from", info["ip"], "lookup", str(info["priority"]), "priority", str(info["priority"]))
        sudoip.run("route", "add", info["subnet"], "dev", nic, "table", str(info["priority"]))
        sudoip.run("route", "add", "default", "via", info["gateway"], "dev", nic, "table", str(info["priority"]))
        


def check_table_creation(winf_names):
    """ Check if the custom route table is created successfully and has content """
    def _check(priority):
        try:
            result = SHELL_POPEN(f'ip route show table {priority}')
            stdout, stderr = result.communicate()
            
            if result.returncode == 0:
                # Check if the table contains any routes (non-empty)
                output = stdout.decode().strip()
                if output:
                    print(f"Table {priority} exists and has content.")
                    return True
                else:
                    print(f"Table {priority} exists but is empty.")
                    return False
            else:
                print(f"Error checking table {priority}: {stderr.decode().strip()}")
                return False
        except Exception as e:
            print(f"Exception while checking table {priority}: {str(e)}")
            return False
        
    initial_priority = 100
    results = []
    for idx in range(len(winf_names)):
        results.append(_check(initial_priority + idx))
        
    return results
    

def clean_up(sudoip:SudoIP, winf_names):
    # Remove all routing rules and routes for the specified interfaces
    initial_priority = 100
    net_addrs = psutil.net_if_addrs()
    nic_info = {}    
    for nic in winf_names:
        nic = nic.strip()
        if not nic:
            continue
        
        addresses = net_addrs[nic]
        # Iterate through each address associated with the interface
        for snicaddr in addresses:
            # Check for IPv4 addresses and retrieve their netmask
            if snicaddr.family == socket.AF_INET:
                nic_info[nic] = {
                    'ip': snicaddr.address,
                    'priority': initial_priority
                }
        initial_priority += 1
            
    for nic in winf_names:
        nic = nic.strip()
        if not nic:
            continue
        
        sudoip.try_run('sudo', 'ip', 'rule', 'del', 'from', f'{nic_info[nic]["ip"]}', 'lookup', f'{nic_info[nic]["priority"]}')
        sudoip.try_run('sudo', 'ip', 'route', 'flush', 'table', f'{nic_info[nic]["priority"]}')

        
        
if __name__ == '__main__':
    import getpass
    password = getpass.getpass("sudo password: ")  # avoids showing it on screen
    sudoip = SudoIP(password)
    winf_names = ['wlx081f7163a93d', 'wlx081f7165e561']
    # seperate_nic(winf_names)
    # clean_up(sudoip, winf_names)
    # print(check_table_creation(winf_names))
    mac_res = check_table_creation(winf_names)
    print(mac_res)
    if not all(mac_res):
        seperate_nic(sudoip, winf_names)
        