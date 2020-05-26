import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def requests_retry_session(retries=3, backoff_factor=0.3, 
                           status_forcelist=(500, 502, 504),
                           session=None):
    """ 
    Retries http access if an error code of 500, 502, or 504 is returned.

    :param int retries: the number of times to try a given request url

    :param backoff_factor: "A backoff factor to apply between attempts after 
                            the second try (most errors are resolved 
                            immediately by a second try without a delay).
                            urllib3 will sleep for:
                            {backoff factor} * (2 ** ({number of total retries} - 1))"
                            source: https://urllib3.readthedocs.io/en/latest/reference/urllib3.util.html

    :param tuple status_forcelist: a tuple of error codes for which the request should
                                   attempt to retry.
    :param requests.Session() session: A standard requests session or an 
                                       OAuth2Session
    :return requests.Session() session: The modified session that is now 
                                        configured to retry multiple times

    Credit: https://www.peterbe.com/plog/best-practice-with-retries-with-requests
    
    """
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session
