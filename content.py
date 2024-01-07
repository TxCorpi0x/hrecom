from sklearn.utils import shuffle
import pandas as pd
import sim
from common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_TEST_SIZE,
    DEFAULT_MIN_SIMILARITY,
)


class ContentBased:
    def __init__(self, item_contents) -> None:
        self.similarity = sim.Similarity()
        self.item_contents = item_contents

    def content_based_similar_items(self, sim_method, items, i, similarity_percent):
        sim_items = []
        for _, item in items.iterrows():
            j = int(item[DEFAULT_ITEM_COL])
            if i == j:
                continue
            genre_sim = self.similarity.get_similarity(
                sim_method, self.item_contents[i], self.item_contents[j]
            )
            if genre_sim > similarity_percent:
                sim_items.append({DEFAULT_ITEM_COL: j, "sim": genre_sim})
        sim_items.sort(key=lambda sim_items: sim_items["sim"], reverse=True)
        df = pd.DataFrame(sim_items)
        return df

    def add_similar_ranking(
        self, sim_method, ratings, items, users, items_to_check, users_to_check
    ):
        new_ratings = ratings.copy()
        shuffle_users = shuffle(users)
        i = 0
        for item_id, mg in self.item_contents.items():
            if i > items_to_check:
                break
            i += 1
            for u_i, u in shuffle_users.iterrows():
                if u_i < users_to_check:
                    curr_rating = ratings[
                        (ratings[DEFAULT_USER_COL] == u[DEFAULT_USER_COL])
                        & (ratings[DEFAULT_ITEM_COL] == item_id)
                    ]

                    if not curr_rating.empty:
                        curr_rating_val = curr_rating.iloc[0][DEFAULT_RATING_COL]
                        if curr_rating_val >= 4.0 or curr_rating_val <= 3.0:
                            sim_items = self.content_based_similar_items(
                                sim_method, items, item_id, DEFAULT_MIN_SIMILARITY
                            )
                            for _, sm in sim_items.iterrows():
                                new_rating = new_ratings[
                                    (
                                        new_ratings[DEFAULT_USER_COL]
                                        == u[DEFAULT_USER_COL]
                                    )
                                    & (
                                        new_ratings[DEFAULT_ITEM_COL]
                                        == sm[DEFAULT_ITEM_COL]
                                    )
                                ]
                                if new_rating.empty:
                                    new_row = {
                                        DEFAULT_USER_COL: u[DEFAULT_USER_COL],
                                        DEFAULT_ITEM_COL: int(sm[DEFAULT_ITEM_COL]),
                                        DEFAULT_RATING_COL: curr_rating_val,
                                        DEFAULT_TIMESTAMP_COL: 1704397300,
                                    }
                                    loc = int(len(new_ratings))
                                    new_ratings.loc[loc] = new_row

        return new_ratings
