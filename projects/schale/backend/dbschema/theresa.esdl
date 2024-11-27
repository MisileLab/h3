module theresa {
  type Letter {
    required name: str {constraint exclusive};
    required tldr: str;
    required file: str;
    multi signers: User;
  }

  type User {
    required name: str;
    required email: str;
    required message: str;
    required hash: str {constraint exclusive};
    signature: str;
  }
}
