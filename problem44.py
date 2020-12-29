from hashlib import sha1
from itertools import combinations
from numtheory import mod_inverse
from operator import itemgetter

from dsa import DSAKeypair, DSASignature, dsa_sign, dsa_verify, hash_msg
from problem43 import get_attack_keypair

p = 0x800000000000000089e1855218a0e7dac38136ffafa72eda7859f2171e25e65eac698c1702578b07dc2a1076da241c76c62d374d8389ea5aeffd3226a0530cc565f3bf6b50929139ebeac04f48c3c84afb796d61e5a4f9a8fda812ab59494232c7d2b4deb50aa18ee9e132bfa85ac4374d7f9091abc3d015efc871a584471bb1
q = 0xf4f47f05794b256174bba6e9b396a7707e563c5b
g = 0x5958c9d3898b224b12672c0b98e06c60df923cb8bc999d119458fef538b8fa4046c8db53039db620c094c9fa077ef389b5322a559946a71903f990f1f7e0e025e2d7f7cf494aff1a0470f5b64c36b625a097f1651fe775323556fe00b3608c887892878480e99041be601a62166ca6894bdd41a7054ec89f756ba9fc95302291
y = 0x2d026f4bf30195ede3a088da85e398ef869611d0f68f0713d51c9c1a3a26c95105d915e2d8cdf26d056b86b8a7b85519b1c23cc3ecdc6062650462e3063bd179c2a6581519f674a61f1d89a1fff27171ebc1b93d4dc57bceb7ae2430f98a6a4d83d8279ee65d71c1203d2c96d65ebbf7cce9d32971c3de5084cce04a2e147821

SIGNATURES = [
    (DSASignature(msg="Listen for me, you better listen for me now. ",
                  s=1267396447369736888040262262183731677867615804316,
                  r=1105520928110492191417703162650245113664610474875),
        'a4db3de27e2db3e5ef085ced2bced91b82e0df19'),
    (DSASignature(msg="Listen for me, you better listen for me now. ",
                  s=29097472083055673620219739525237952924429516683,
                  r=51241962016175933742870323080382366896234169532),
        'a4db3de27e2db3e5ef085ced2bced91b82e0df19'),
    (DSASignature(msg="When me rockin' the microphone me rock on steady, ",
                  s=277954141006005142760672187124679727147013405915,
                  r=228998983350752111397582948403934722619745721541),
        '21194f72fe39a80c9c20689b8cf6ce9b0e7e52d4'),
    (DSASignature(msg="Yes a Daddy me Snow me are de article dan. ",
                  s=1013310051748123261520038320957902085950122277350,
                  r=1099349585689717635654222811555852075108857446485),
        '1d7aaaa05d2dee2f7dabdc6fa70b6ddab9c051c5'),
    (DSASignature(msg="But in a in an' a out de dance em ",
                  s=203941148183364719753516612269608665183595279549,
                  r=425320991325990345751346113277224109611205133736),
        '6bc188db6e9e6c7d796f7fdd7fa411776d7a9ff'),
    (DSASignature(msg="Aye say where you come from a, ",
                  s=502033987625712840101435170279955665681605114553,
                  r=486260321619055468276539425880393574698069264007),
        '5ff4d4e8be2f8aae8a5bfaabf7408bd7628f43c9'),
    (DSASignature(msg="People em say ya come from Jamaica, ",
                  s=1133410958677785175751131958546453870649059955513,
                  r=537050122560927032962561247064393639163940220795),
        '7d9abd18bbecdaa93650ecc4da1b9fcae911412'),
    (DSASignature(msg="But me born an' raised in the ghetto that I want yas to know, ",
                  s=559339368782867010304266546527989050544914568162,
                  r=826843595826780327326695197394862356805575316699),
        '88b9e184393408b133efef59fcef85576d69e249'),
    (DSASignature(msg="Pure black people mon is all I mon know. ",
                  s=1021643638653719618255840562522049391608552714967,
                  r=1105520928110492191417703162650245113664610474875),
        'd22804c4899b522b23eda34d2137cd8cc22b9ce8'),
    (DSASignature(msg="Yeah me shoes a an tear up an' now me toes is a show a ",
                  s=506591325247687166499867321330657300306462367256,
                  r=51241962016175933742870323080382366896234169532),
        'bc7ec371d951977cba10381da08fe934dea80314'),
    (DSASignature(msg="Where me a born in are de one Toronto, so ",
                  s=458429062067186207052865988429747640462282138703,
                  r=228998983350752111397582948403934722619745721541),
        'd6340bfcda59b6b75b59ca634813d572de800e8f'),
]

def test_hash_function_correct():
    keypair = DSAKeypair(y, None, g, p , q)
    for sig, hash in SIGNATURES:
        assert hash_msg(sig.msg) == int(f'0x{hash}', 16)
        dsa_verify(sig, keypair)


def test_find_repeated_nonce():
    # This is just a brute force attack
    for sig1, sig2 in combinations(map(itemgetter(0), SIGNATURES), 2):
        assert sig1 != sig2
        s_inv = mod_inverse(sig1.s - sig2.s, q)
        k = ((hash_msg(sig1.msg) - hash_msg(sig2.msg)) * s_inv) % q
        attack_keypair = get_attack_keypair(sig1.msg, sig1, y, k)
        attack_sig = dsa_sign(sig1.msg, attack_keypair, k)
        if attack_sig == sig1:
            secret_str = '{:02x}'.format(attack_keypair.private)
            hash_obj = sha1(secret_str.encode('utf-8'))
            digest = hash_obj.hexdigest()
            assert digest == 'ca8f6f7c66fa362d40760d135b763eb8527d3d52'
            break
