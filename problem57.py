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


def bob_encrypt(h):
    k = mod_exp(h, bob_secret, p)
    h = sha1("crazy flamboyant for the rap enjoyment".encode())
    h.update(str(k).encode())
    return h.digest()


def find_element_of_order(r):
    # find an element of order r mod p
    h = 1
    while h == 1:
        h = mod_exp(randint(1, p), (p - 1) // r, p)

    assert mod_exp(h, r, p) == 1, 'Element h should have been h^r = 1 mod p'

    return h


def brute_force_digest(digest, h, r):
    assert mod_exp(h, r, p) == 1, 'Element h should have had order r'
    # h has order r so the digest must have value h^x where 0 <= x < r - 1
    # Must include x = r (so K would be 1)
    for x in range(1, r + 1):
        d = sha1('crazy flamboyant for the rap enjoyment'.encode())
        d.update(str(mod_exp(h, x, p)).encode())
        candidate_digest = d.digest()

        if digest == candidate_digest:
            return x

    raise ValueError('Could not find h^x such that h^x = K mod p')

if __name__ == '__main__':
    crt_moduli = []
    running_total = 1
    for r in small_factors(j):
        h = find_element_of_order(r)
        digest = bob_encrypt(h)
        b = brute_force_digest(digest, h, r)
        crt_moduli.append((b, r))
        running_total *= r
        if running_total > q:
            break

    assert bob_secret == crt_inductive(crt_moduli)
    print(f'All done!  Successfully cracked bob_secret using {len(crt_moduli)} samples')
