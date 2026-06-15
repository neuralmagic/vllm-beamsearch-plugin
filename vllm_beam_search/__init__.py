def __getattr__(name: str):
    if name == "BeamSearchScheduler":
        from vllm_beam_search.scheduler import BeamSearchScheduler

        return BeamSearchScheduler
    raise AttributeError(name)


def register_beam_search_plugin() -> None:
    from vllm_beam_search.scheduler import _install_worker_history_rewrite_hooks

    _install_worker_history_rewrite_hooks()


__all__ = ["BeamSearchScheduler", "register_beam_search_plugin"]
