# Regular Expression

```python
import re
```

pattern = r"xxxx"

pattern = r"a|b" (multiple conditions)

eg. pattern = r"run|ran" = pattern = r"r[ua]n"

eg. pattern = r"found|find" = pattern = r"f(ou|i)nd"

| Mark  | Meaning                      | Range                       |
| ----- | ---------------------------- | --------------------------- |
| \d    | number                       | [0-9]                       |
| \D    | not number                   |                             |
| \s    | any whitespace character     | [\t\\n\r\f\v]               |
| \S    | not whitespace character     |                             |
| \w    | any letter,number and _      | [a-zA-Z0-9_]                |
| \W    | not \w                       |                             |
| \b    | match a word margin          | eg. er\b-> never ok, verb × |
| \B    | match a not word magin       | eg. er\B-> never ×, verb ok |
| \\\   | \                            |                             |
| .     | any character except \n      |                             |
| ?     | preceding modes are optional |                             |
| *     | repeat 0-multiple times      |                             |
| +     | repeat 1-multiple times      |                             |
| {n,m} | repeat n-m times             |                             |
| {n}   | repeat n times               |                             |
| +?    | match + at least one time    |                             |
| **?*  |                              |                             |
| ??    |                              |                             |
|       |                              |                             |
|       |                              |                             |
|       |                              |                             |

re functions:

| Function      | Description                                         | eg.                                                                                                |
| ------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| re.search()   | scan the whole string, find first matched mode      | re.search(r"run", "I run to you") >'run'                                                           |
| re.match()    | match from the head                                 | re.search(r"run", "I run to you") >None                                                            |
| re.findall()  | return unrepeated pattern list                      | re.findall(r"r[ua]n", "I run to you. you ran to him") > ['run', 'ran']                             |
| re.finditer() | same as findall but used for iteration              |                                                                                                    |
| re.split()    | use re to split string                              | re.split(r"r[ua]n","" I run to you. you ran to him") > ['I ', ' to you. you ', ' to him']          |
| re.sub()      | use re to replace string                            | re.sub(r"r[ua]n", "jump", "I run to you. you ran to him") > 'I jump to you. you jump to him'       |
| re.subn()     | same as sub but return replacing times additionally | re.subn(r"r[ua]n", "jump", "I run to you. you ran to him") > ('I jump to you. you jump to him', 2) |

flags

| mode | full name    | explannation |
| ---- | ------------ | ------------ |
| re.I | re.IGNORCASE |              |
| re.M | re.MULTILINE |              |
| re.S | re.DOTALL    |              |
| re.L | re.LOCALE    |              |
| re.U | re.UNICODE   |              |
| re.X |              |              |






