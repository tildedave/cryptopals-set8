from random import randint
from typing import Tuple


from elliptic_curve import EllipticCurve, EllipticCurvePoint


class DiffieHelman:
    def __init__(self, curve: EllipticCurve, point: EllipticCurvePoint):
        self.curve = curve
        self.point = point

    def generate_keypair(self,
                         given_order: int,
                         ) -> Tuple[int, EllipticCurvePoint]:
        """
        Generate an ECDH keypair.

        Alice and Bob agree to use the given curve and start with a given point on
        a curve.  (Public information)

        Alice and Bob both generate a random integer between 0 and the order of the
        point on the curve. (Secret information)

        Diffie-Hellmen then works as follows:
        Alice keypair: (secret_A, g^secret_A)
        Bob keypair: (secret_B, g^secret_B)

        Alice and Bob then create the shared key:
            (g^secret_A)^(secret_B) = (g^secret_B)^(secret_A)
        """
        secret = randint(0, given_order)
        public = self.curve.scalar_mult(self.point, secret)

        return (secret, public)

    def compute_secret(self,
                       peer_public: EllipticCurvePoint,
                       self_secret: int) -> EllipticCurvePoint:
        """
        Compute the public key ^ secret.
        """
        return self.curve.scalar_mult(peer_public, self_secret)
