"""Shared memory workspace demonstration."""

from vajra_agent import SharedMemory


def main():
    shared = SharedMemory()
    shared.publish_observation(
        "ArchitectAgent", "Architectural plan finalized: Microservices pattern selected."
    )
    shared.publish_observation("CoderAgent", "Implemented API gateway in api/main.py")

    print("Shared Team Observations:")
    for obs in shared.shared_observations:
        print(f"  [{obs['sender']}]: {obs['text']}")


if __name__ == "__main__":
    main()
