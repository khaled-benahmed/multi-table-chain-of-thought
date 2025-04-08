#############
ChainOfTables
#############
Use language models to automatically process tabular data iteratively.

***************
Hints
***************

Using .env file, one can change the LLM used to process data.

***************
Pre-commit hook
***************

To ensure minimal code quality, a pre-commit (https://pre-commit.com/) hook is added to the repository. To enable the commit hook, execute:

.. code-block:: shell

    pre-commit install

********************
Deployment (on lyra)
********************

The compose file can be used for deployment. If you are planning to deploy to our GPU System (lyra), it is possible to connect to its docker using

.. code-block:: shell

    docker context create remote-lyra --docker "host=ssh://<user-name-on-lyra>@lyra.dbs.uni-hannover.de"
    docker context use remote-lyra

deployment can be done either locally or using the remote docker context by running 

.. code-block:: shell

    docker compose -f compose-textgen.yaml -f compose-tables-chain.yaml up
