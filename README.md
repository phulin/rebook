# Dewarp
### forked from phulin/rebook. Refer to that Readme for more info of this repo.

## This is a modified version that is a pip installable with command:
```
pip install https://github.com/pibit-ai/dewarp/archive/refs/heads/master.zip
```

```python
from dewarp.dewarp import fix_dewarp
import cv2

warped_img = cv2.imread("<img path>")
dewarped_img = fix_dewarp(warped_img)

```

## Dewarping

`dewarp.py` contains implementations of two dewarping algorithms:

* [Kim et al. 2015, Document dewarping via text-line based optimization](http://www.sciencedirect.com/science/article/pii/S003132031500165X)
* [Meng et al. 2011, Metric rectification of curved document images](http://ieeexplore.ieee.org/abstract/document/5975161/)

Focal length is currently assumed to be that of the iPhone 7, because that’s what I have been using to test. Change the f value at the top of this file if using a different camera.
