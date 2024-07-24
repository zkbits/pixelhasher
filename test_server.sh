#!/bin/bash

# run this script to simulate a pool server. make your config.json like this:
#
#
# {
#   "payout_address": "0x1000000000000000000000000000000000000001",
#   "server_name": "127.0.0.1",
#   "server_port": 17890
# }

while sleep 1; do (echo '{"method": "set_work", "pool_address": "0xfaf20e5ca7e39d43a3aabc450602b4147c3aa62e", "challenge_number": "0x2222222222222222222222222222222222222222222222222222222222222222", "mining_target": "0x00000000bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "sprite": "0x00000000000007e005a03ffc27e427e437ec37ec07e006600660000000000000"}') | nc -l 17890; done
