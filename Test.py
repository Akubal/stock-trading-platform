from nsepy import get_history
from datetime import date
from nsetools import Nse

nse = Nse()
q = nse.get_quote('TATAMOTORS')
print(q['lastPrice'])
