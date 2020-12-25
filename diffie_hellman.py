from random import randint
from typing import NamedTuple


from elliptic_curve import EllipticCurve, EllipticCurvePoint


class ECDHConfig(NamedTuple):
    curve: EllipticCurve
    point: EllipticCurvePoint
    n: int  # this is the point order

    def rand_scalar(self) -> int:
        return randint(1, self.n)

    def scalar_mult_point(self, k: int) -> EllipticCurvePoint:
        return self.curve.scalar_mult(self.point, k)

    def is_valid(self):
        pt = self.curve.scalar_mult(self.point, self.n)
        return self.curve.is_identity(pt)


class ECDHKeypair(NamedTuple):
    config: ECDHConfig
    secret: int
    public: EllipticCurvePoint

    def compute_secret(self,
                       peer_public: EllipticCurvePoint,
                       ) -> EllipticCurvePoint:
        """
        Compute the public key ^ secret.
        """
        return self.config.curve.scalar_mult(peer_public, self.secret)

    def is_valid(self) -> bool:
        public = self.config.curve.scalar_mult(self.config.point, self.secret)
        return self.public == public


class DiffieHellman:
    def __init__(self,
                 curve: EllipticCurve,
                 point: EllipticCurvePoint,
                 point_order: int,
                 ):
        self.config = ECDHConfig(curve, point, point_order)

    def generate_keypair(self) -> ECDHKeypair:
        """
        Generate an ECDH keypair.

        Alice and Bob agree to use the given curve and start with a given point
        on a curve.  (Public information)

        Alice and Bob both generate a random integer between 0 and the order of
        the point on the curve. (Secret information)

        Diffie-Hellmen then works as follows:
        Alice keypair: (secret_A, g^secret_A)
        Bob keypair: (secret_B, g^secret_B)

        Alice and Bob then create the shared key:
            (g^secret_A)^(secret_B) = (g^secret_B)^(secret_A)
        """
        secret = self.config.rand_scalar()
        public = self.config.scalar_mult_point(secret)

        return ECDHKeypair(self.config, secret, public)
