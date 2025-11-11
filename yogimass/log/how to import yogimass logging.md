```python
from log import yogimass_logging
log_formatter, logger = yogimass_logging.main()
logger.debug(f'Debug notes for {__name__}')
```