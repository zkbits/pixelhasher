// input to keccak(...) is 84 bytes: challenge|sender|nonce == 32+20+32 bytes
#define LEN84_INPUT 84
#define LEN32_CHNUM 32 // 32 bytes: challenge_number
#define LEN20_SENDR 20 // 20 bytes: message.sender/miner_address/pool_address
#define LEN32_NONCE 32 // 32 bytes: nonce (which contains zkbit and entropy)
#define LEN32_MTRGT 32 // 32 byets: mining_target
#define LEN32_KSHA3 32 // 32 bytes: keccak256() output length
