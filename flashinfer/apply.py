try:
    from flashinfer_bench.apply import apply as apply
except ImportError:
    def apply(fn):
        print("Flashinfer Bench is not available")
        def decorator(f):
            return f
        return decorator