# Authentication Troubleshooting

## Login Token Decode Errors

If login fails with `JWTDecodeError`, check that the API and frontend use the
same signing secret and algorithm. Previous fixes rotated the stale local
secret, cleared old browser tokens, and restarted the model server so token
validation used the current environment.

## Local Development

Run the model server on port 8001 and keep API credentials in environment
variables. Do not commit local secrets.
