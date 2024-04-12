module default {
  type KeyStore {
    name: str;
    pubkey: str;
    privkey: str;
  }

  type User {
    userid: str;
    admin: bool;
    multi groups: KeyStore {
      name: str;
      pubkey: str;
    }
  }
}
