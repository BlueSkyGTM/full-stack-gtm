## Ship It

Production tool loops fail in three predictable ways. The model emits malformed JSON for arguments. The model calls a function that doesn't exist in your registry. The function itself raises an exception — network timeout, rate limit, bad input. Each failure must return a structured error to the model