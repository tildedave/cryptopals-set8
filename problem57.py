from random import randint
from hashlib import sha1
from numtheory import small_factors, mod_exp, euclid_extended, crt_inductive

p = 7199773997391911030609999317773941274322764333428698921736339643928346453700085358802973900485592910475480089726140708102474957429903531369589969318716771
g = 4565356397095740655436854503483826832136106141639563487732438195343690437606117828318042418238184896212352329118608100083187535033402010599512641674644143
# order of g is q
q = 236234353446506858198510045061214171961

bob_secret = randint(0, q)

# Alice and Bob choose secret keys as random integers mod q
# j = (p - 1) / q
j = 30477252323177606811760882179058908038824640750610513771646768011063128035873508507547741559514324673960576895059570


def encrypt_digest(h, secret, p):
    k = mod_exp(h, secret, p)
    h = sha1("crazy flamboyant for the rap enjoyment".encode())
    h.update(str(k).encode())
    return h.digest()


def find_element_of_order(r, p):
    # find an element of order r mod p
    h = 1
    while h == 1:
        h = mod_exp(randint(1, p), (p - 1) // r, p)

    assert mod_exp(h, r, p) == 1, 'Element h should have been h^r = 1 mod p'

    return h


def brute_force_digest(digest, h, r, p):
    assert mod_exp(h, r, p) == 1, 'Element h should have had order r'
    # h has order r so the digest must have value h^x where 0 <= x < r - 1
    # Must include x = r (so K would be 1)
    for x in range(1, r + 1):
        d = sha1('crazy flamboyant for the rap enjoyment'.encode())
        d.update(str(mod_exp(h, x, p)).encode())
        candidate_digest = d.digest()

        if digest == candidate_digest:
            return x

    raise ValueError(f'Could not find h^x such that h^x = K mod p')


def find_residues(j, p, secret):
    for r in small_factors(j):
        h = find_element_of_order(r, p)
        # Simulates getting a message from Bob using the secret (h)
        digest = encrypt_digest(h, secret, p)
        b = brute_force_digest(digest, h, r, p)
        yield (b, r)

if __name__ == '__main__':
    crt_moduli = list(find_residues(j, p, bob_secret))
    x, _ = crt_inductive(crt_moduli)
    assert bob_secret == x
    print(f'All done!  Successfully cracked bob_secret using {len(crt_moduli)} samples')
