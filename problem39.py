import random
from typing import NamedTuple

from rsa import RSAKeypair, rsa_decrypt, rsa_encrypt

def test_create_keypair():
    keypair = RSAKeypair.create(e=3)
    assert rsa_decrypt(rsa_encrypt("bob", keypair), keypair) == "bob"
