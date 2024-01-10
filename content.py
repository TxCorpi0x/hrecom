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
    DEFAULT_SIMILARITY_COL,
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
        new_ratings[DEFAULT_ITEM_COL] = new_ratings[DEFAULT_ITEM_COL].astype("int64")
        shuffle_users = shuffle(users)
        i = 0
        for item_id, mg in self.item_contents.items():
            if i > items_to_check:
                break
            i += 1
            for u_i, u in shuffle_users.iterrows():
                added_rates = 0
                if added_rates < users_to_check:
                    curr_rating = new_ratings[
                        (new_ratings[DEFAULT_USER_COL] == u[DEFAULT_USER_COL])
                        & (new_ratings[DEFAULT_ITEM_COL] == item_id)
                    ]

                    if not curr_rating.empty:
                        curr_rating_val = curr_rating.iloc[0][DEFAULT_RATING_COL]
                        if curr_rating_val >= 3.0:
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
                                    # update counter
                                    added_rates += 1

        return new_ratings

    def get_items_sim_merged(self, sim_method, content_col, sim_col, splitter):
        cross_col = "key"
        self.item_contents[cross_col] = 1
        items_sim = self.item_contents.merge(self.item_contents, on=cross_col)
        col1 = content_col + "_x"
        col2 = content_col + "_y"
        items_sim[sim_col] = items_sim.apply(
            lambda i: self.similarity.get_similarity(
                sim_method,
                str(i[col1]).split(splitter),
                str(i[col2]).split(splitter),
            ),
            axis=1,
        )
        return items_sim.drop([col1, col2, cross_col], axis=1)

    def get_new_ratings(
        self, movies_sim_reduced_top, movies_users_ratings_high, new_users_rate_count=10
    ):
        col_original = DEFAULT_ITEM_COL + "_x"
        col_similar = DEFAULT_ITEM_COL + "_y"
        new_ratings = (
            movies_users_ratings_high.merge(
                # filter the actual rate itself
                movies_sim_reduced_top[
                    movies_sim_reduced_top[col_original]
                    != movies_sim_reduced_top[col_similar]
                ],
                left_on=[DEFAULT_ITEM_COL],
                right_on=[col_original],
            )
            .drop(
                [DEFAULT_ITEM_COL, col_original, DEFAULT_SIMILARITY_COL],
                axis=1,
            )
            .groupby(DEFAULT_USER_COL)
            .head(new_users_rate_count)
        )
        return new_ratings.rename(columns={col_similar: DEFAULT_ITEM_COL})
