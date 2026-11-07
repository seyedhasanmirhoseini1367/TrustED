from django import template

register = template.Library()


@register.filter
def pct(x, digits=1):
    if x is None or x == "":
        return "—"
    return f"{float(x) * 100:.{int(digits)}f} %"


@register.filter
def num(x, digits=3):
    if x is None or x == "":
        return "—"
    return f"{float(x):.{int(digits)}f}"


@register.filter
def get(d, key):
    return d.get(key) if isinstance(d, dict) else None


@register.filter
def joinlist(v):
    return ", ".join(map(str, v)) if v else "—"


@register.filter
def mul100(x):
    if x is None or x == "":
        return ""
    return f"{float(x) * 100:g}"
