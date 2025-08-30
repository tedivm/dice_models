# Best Practices

## General

* Assume the minimum version of Python is 3.10.
* Prefer async libraries and functions over synchronous ones.
* Always define dependencies and tool settings in the `pyproject.toml` file.
* Prefer using existing dependencies when possible over adding new ones.
* For complex code always consider using a third party library instead of maintaining your own logic.
* Never use `setup.py` or `setup.cfg` files.
* When calling functions and methods, keyword arguments should be used instead of positional arguments.
* Avoid using global instances of classes or data, instead rely on dependency injection.
* Do not use deprecated functions or libraries.

## Production Ready

* All generated code should be production ready.
* There should be no stubs "for production".
* Any code or package differences between Development and Production should be avoided unless absolutely necessary.

## Logging

* Do not use `print` for logging or debugging, use the `getLogger` logger instead.
* Each file should get its own logger using the `__name__` variable for a name.
* Logging levels should be used to allow developers to enable richer logging levels while testing than they would want in production.
* Most caught exceptions should be logged with a `logger.exception` call.

## Commenting

* Comments should improve the readability and understandability of the code.
* Comments should not simply exist for the sake of existing.
* Examples of things that should be commented on are unclear function names/parameters, decisions about the choice of settings or functions used, descriptions of logic used or why a variable was defined, and other things that allow developers to properly follow the code.
* Other examples include things such as security risks, places where code may cover edge cases, and advice for developers who are refactoring or expanding the code.
* Comments should be concise and accurate.
* Comments should provide value and improve the codebase.

## Error Handling

* Do not suppress exceptions unless the exception is expected, and make sure to handle the exception properly when it is.
* When suppressing exceptions make sure to log them using `logger.exception`.

## Security

* Always write secure code.
* Never hardcode sensitive data.
* Do not log sensitive data.
* All user input should be validated.

## Typing

* Everything should be typed: function signatures (including return values), variables, and anything else.
* Use the union operator to specify when multiple types are allowed.
* Do not use `Optional`, use a union with `None` (ie, `str|None`).
* Use the metaclasses from the typing library instead of native types for objects and lists (ie, `Dict[str, str]` and `List[str]` instead of `dict` or `list`).
* Avoid using `Any` unless absolutely necessary.
* If the schema is defined use a `dataclass` with properly typed parameters instead of a `dict`.

## Settings

* Applications settings should be managed with the `pydantic-settings` library.
* Sensitive configuration data should always use the Pydantic `SecretStr` or `SecretBytes` types.
* Settings that are allowed to be unset should default to `None` instead of empty strings.
* Settings should be defined with the Pydantic `Field` function, and include descriptions for users.
* Use `field_validator` instead of `validator` for all Pydantic models.

## FastAPI

* APIs should adhere as closely as possible to REST principles, including the use of GET/PUT/POST/DELETE HTTP Verbs in their appropriate way.
* All routes should use Pydantic models for input and output models.
* Different Pydantic models should be used for Inputs and Outputs (ie, creating a `Post` should require a `PostCreate` and return a `PostRead` model, not reuse the same model for each).
* The parameters in pydantic models for user input should use the Field function and have validation and descriptions.

## SQLAlchemy

* Always use the async SQLAlchemy APIs with the SQLAlchemy 2.0 syntax.
* Database tables should be represented with the declarative class system.
* Alembic should be used to define migrations.
* Migrations should be compatible with both SQLite and PostgreSQL.
* When creating queries do not use implicit `and`, instead use the `and_` function (instead of `where(Model.parameter_a == A, Model.parameter_b == B)` do `where(and_(Model.parameter_a == A, Model.parameter_b == B))`).

## Typer

* Any CLI command or script that should me made accessible to users should be exposed via the Typer library.
* The main entrypoint for the cli should be `PACKAGE_NAME/cli.py`.

## Testing

* Do not wrap test functions in classes unless there is a specific technical reason to do so, and instead prefer single functions.
* Make sure all fixtures are defined or imported into the `conftest.py` file so they are available to all tests.
* Do not use mocks to replace simple dataclasses or Pydantic models unless absolutely necessary to write the test, instead simply create an instance of the appropriate class with the desired parameters.
* Use the FastAPI Test Client (preferably with a fixture to generate it) rather than calling FastAPI router classes directly.
* Use a test database fixture with SQLite backed by memory for tests that require a database. Including a dependency override for this test database as part of the FastAPI App Fixture is extremely useful.
* When adding new code you should also add the appropriate tests to cover that new code.

## Files

* Filenames should always be lower case for better compatibility with case insensitive filesystems.
* This includes the names of documentation files, with the exception of standard files (like `README.md`, `LICENSE`, etc).
* Developer documentation should live in `docs/dev`.
* New developer documents should be added to the table of contents in the `docs/dev/README.md` file.
* Files only meant for use with building containers should live in the `docker/` folder.
* Database models should live in `PACKAGE_NAME/models/`.
* The primary settings file should live in `PACKAGE_NAME/conf/settings.py`.

## Developer Environments

* Developers should always be able to start a fully functional developer instance of an application with `docker compose up`.
* Developer environments should be initialized with fake data for easy developer use.
* Developer settings should live in the `.env` file, which should be in the `.gitignore` file.
* A `.env.example` environment file should exist as a template for new developers to create their `.env` file. This file can be read to learn what variables should be set.
