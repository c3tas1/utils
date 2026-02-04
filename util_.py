def find_longest_match(pairs, search_list):
    best = []
    best_pair_start = 0
    best_search_start = 0

    for si in range(len(search_list)):
        for pi in range(len(pairs)):
            picks = []
            s, p = si, pi
            while p < len(pairs) and s < len(search_list):
                found = False
                for val in pairs[p]:
                    if val == search_list[s]:
                        picks.append(val)
                        s += 1
                        found = True
                        break
                if not found:
                    break
                p += 1
            if len(picks) > len(best):
                best = picks[:]
                best_pair_start = pi
                best_search_start = si

    return {
        "match": best,
        "length": len(best),
        "pair_start_index": best_pair_start,
        "search_start_index": best_search_start,
    }


# --- Example ---
pairs = [
    [1.21, 58.49],
    [68.6, 5.49],
    [5.49, 1.14],
    [1.1, 6.49, 9.89],
    [9.99, 97.9],
]

search_list = [4.59, 8.59, 5.49, 6.49, 5.47, 97.8]

result = find_longest_match(pairs, search_list)
print(result)
