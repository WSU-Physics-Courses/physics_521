# physics_521
Physics 521: Classical Mechanics

# Usage

Include something like the following at the top of your CoLaboratory
notebooks to clone, update, and install the modules:

```python
from IPython.display import clear_output
!git clone https://github.com/mforbes/physics_521.git
!cd physics_521 && git pull
try: from physics_521 import nbinit; clear_output()
except: raise
```
