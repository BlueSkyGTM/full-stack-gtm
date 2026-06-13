# Editor Setup

## Hook

You'll spend thousands of hours in your editor. A misconfigured one will slow you down at every step—from writing prompts to debugging API responses. This lesson covers the configurations that matter for AI engineering work specifically.

## Concept

LSP integration, linting for Python/TypeScript, JSON schema validation for API payloads, and keybinding workflows that reduce round-trips between editor and terminal. Mechanism: how language servers analyze code, how formatters enforce consistency, and how schema validation catches malformed API calls before they hit a remote endpoint.

## Examples

Three runnable demonstrations: (1) a Python file that passes linting and formatting checks, then prints confirmation; (2) a JSON payload validated against a schema, with output showing pass/fail; (3) a shell command that installs extensions and prints their status to stdout.

## Use It

Clean API payloads and consistent formatting reduce debugging time when building GTM integrations. Specifically: JSON schema validation catches malformed Clay webhook payloads before submission. This is foundational for Zone I and Zone II—any workflow involving API calls to enrichment or scoring endpoints requires a correctly configured editor to iterate quickly.

## Ship It

**Easy:** Configure your editor to format Python on save, run it against a provided script, and print "formatted" to confirm.  
**Medium:** Write a JSON schema for a generic enrichment API response, validate a sample payload against it, and print the validation result.  
**Hard:** Create a multi-file project (Python module + JSON config), configure linting + formatting + schema validation to run as a single command, and print a combined pass/fail summary.

## Evaluate

Questions target: which editor features catch errors before runtime vs. at runtime, how LSP integration changes the edit-test cycle, and what specific configuration prevents malformed API payloads in GTM workflows. Grounded in the mechanisms covered in Concept and Examples.