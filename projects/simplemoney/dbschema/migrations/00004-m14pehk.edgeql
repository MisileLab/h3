CREATE MIGRATION m14pehktqebicxbpceu7fctj5djnyxgzhh2u6tl3jsygqgft6xhbka
    ONTO m1yw5cr2t3s4dprwgdvt5fus22izda4w7cpxgmb3gji3nsckekbcda
{
  ALTER TYPE default::Transaction {
      ALTER PROPERTY sent {
          RENAME TO to;
      };
  };
};
