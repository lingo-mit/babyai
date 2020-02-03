COLORS = ['yellow', 'green', 'blue', 'purple', 'green', 'red', 'grey', '']
KEYS = ['{} key'.format(color) for color in COLORS]
BLOCKS = ['{} box'.format(color) for color in COLORS]
BALLS = ['{} ball'.format(color) for color in COLORS]
DOORS = ['{} door'.format(color) for color in COLORS]
OBJECTS = KEYS + BLOCKS + BALLS 
THE_OBJECTS = ['the {}'.format(obj) for obj in OBJECTS]
A_OBJECTS = ['a {}'.format(obj) for obj in OBJECTS]
THE_DOORS = ['the {}'.format(obj) for obj in DOORS]
A_DOORS = ['a {}'.format(obj) for obj in DOORS]
ALL_OBJECTS = THE_OBJECTS + A_OBJECTS
ALL_DOORS = THE_DOORS + A_DOORS
ALL_OBJECTS2 = []

for obj in ALL_OBJECTS:
    new = ' '.join(obj.split())
    ALL_OBJECTS2.append(new) 
ALL_OBJECTS = ALL_OBJECTS2

ALL_DOORS2 = []
for obj in ALL_DOORS:
    new = ' '.join(obj.split())
    ALL_DOORS2.append(new) 
ALL_DOORS = ALL_DOORS2 


REL_POS = ['in front of you', 'behind you', 'on your right', 'on your left']
STARTERS = ['pick up', 'put', 'go to', 'open']
EVERYTHING = REL_POS + ALL_OBJECTS + ALL_DOORS
TEMPLATES = {'put': 'put the [MASK] next to the [MASK]', 'pick': 'pick up the [MASK]', 'open': 'open the [MASK]', 'go': 'go to the [MASK]'}
MAPPINGS = {'door': ALL_DOORS, 'obj': ALL_OBJECTS, 'pos': REL_POS, 'start': STARTERS}

INVERSE_TEMPLATES = {}
for name, temp in TEMPLATES.items(): 
    INVERSE_TEMPLATES[temp] = name

DETAILED_TEMPLATES = {'put': 'put [obj] next to [obj|door] [?pos]', 
                      'go': 'go to [obj] [?pos]', 'open': 'open [door] [?pos]', 'pick': 'pick up [obj] [?pos]'}

# Notes: 
# input to grammar: dict of template name --> template with holes. holes should refer to filler classes. 
# ? in hole indicates that the component is optional 
# A | B in hole indicates that either A or B can fill that hole 

# define mappings, detailed templates, templates, inverse templates (can simplify
# the template stuff)