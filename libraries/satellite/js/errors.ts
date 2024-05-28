function safeCaptureStackTrace(err: Error, constructorOpt: unknown) {
  if ('captureStackTrace' in Error) {
    // @ts-ignore
    Error.captureStackTrace(err, constructorOpt);
  } else {
    // Fallback to manual stack trace construction
    const stack = new Error().stack;
    err.stack = stack? stack : '';
  }
}

export class nullValue extends Error {
  constructor() {
    super("nullValue");
    safeCaptureStackTrace(this, nullValue);
  }
}

export class NaNValue extends Error {
  constructor() {
    super("NaNValue");
    safeCaptureStackTrace(this, NaNValue);
  }
}

