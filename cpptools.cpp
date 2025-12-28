#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <set>

namespace py = pybind11;
using namespace std;

const int BOARD_SIZE = 19;

inline bool valid_pos(int x, int y) {
    return x >= 0 && y >= 0 && x < BOARD_SIZE && y < BOARD_SIZE;
}

void remove_die(vector<vector<int>> &board) {
    // do not have already die on board
    // how to determine almost die
    return;
}

bool value_situation(py::array_t<int> board_in) {
    auto buf = board_in.request();
    if (buf.ndim != 3) throw std::runtime_error("Number of dimensions must be 3");
    // evaluate board precisely
    auto ptr = static_cast<int*>(buf.ptr);
    auto s0 = buf.strides[0] / sizeof(int);
    auto s1 = buf.strides[1] / sizeof(int);
    auto s2 = buf.strides[2] / sizeof(int);
    // check value of s0, s1, s2
    vector<vector<int>> mboard(BOARD_SIZE, vector<int>(BOARD_SIZE, 0));
    vector<vector<int>> mboard2(BOARD_SIZE, vector<int>(BOARD_SIZE, 0));

    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            int white_piece = ptr[0 * s0 + i * s1 + j * s2];
            int black_piece = ptr[1 * s0 + i * s1 + j * s2];
            if (black_piece > 0) {
                mboard[i][j] = 128;
            } else if (white_piece > 0) {
                mboard[i][j] = -128;
            }
        }
    }
            
    remove_die(mboard);

    // Bouzy's 5/21 Algorithm
    // 5 Dilation
    for (int t = 0; t < 5; t++) {
        mboard2 = mboard;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (mboard2[i][j] > 0) {
                    int count = 0;
                    if (valid_pos(i - 1, j)) {
                        if (mboard2[i - 1][j] < 0) {
                            continue;
                        }
                        if (mboard2[i - 1][j] > 0) {
                            count++;
                        }
                    }
                    if (valid_pos(i + 1, j)) {
                        if (mboard2[i + 1][j] < 0) {
                            continue;
                        }
                        if (mboard2[i + 1][j] > 0) {
                            count++;
                        }
                    }
                    if (valid_pos(i, j - 1)) {
                        if (mboard2[i][j - 1] < 0) {
                            continue;
                        }
                        if (mboard2[i][j - 1] > 0) {
                            count++;
                        }
                    }
                    if (valid_pos(i, j + 1)) {
                        if (mboard2[i][j + 1] < 0) {
                            continue;
                        }
                        if (mboard2[i][j + 1] > 0) {
                            count++;
                        }
                    }
                    mboard[i][j] += count;
                } else if (mboard[i][j] < 0) {
                    int count = 0;
                    if (valid_pos(i - 1, j)) {
                        if (mboard2[i - 1][j] > 0) {
                            continue;
                        }
                        if (mboard2[i - 1][j] < 0) {
                            count++;
                        }
                    }
                    if (valid_pos(i + 1, j)) {
                        if (mboard2[i + 1][j] > 0) {
                            continue;
                        }
                        if (mboard2[i + 1][j] < 0) {
                            count++;
                        }
                    }
                    if (valid_pos(i, j - 1)) {
                        if (mboard2[i][j - 1] > 0) {
                            continue;
                        }
                        if (mboard2[i][j - 1] < 0) {
                            count++;
                        }
                    }
                    if (valid_pos(i, j + 1)) {
                        if (mboard2[i][j + 1] > 0) {
                            continue;
                        }
                        if (mboard2[i][j + 1] < 0) {
                            count++;
                        }
                    }
                    mboard[i][j] -= count;
                } else {
                    int countp = 0;
                    int countn = 0;
                    if (valid_pos(i - 1, j)) {
                        if (mboard2[i - 1][j] < 0) {
                            countn++;
                        }
                        if (mboard2[i - 1][j] > 0) {
                            countp++;
                        }
                    }
                    if (valid_pos(i + 1, j)) {
                        if (mboard2[i + 1][j] < 0) {
                            countn++;
                        }
                        if (mboard2[i + 1][j] > 0) {
                            countp++;
                        }
                    }
                    if (valid_pos(i, j - 1)) {
                        if (mboard2[i][j - 1] < 0) {
                            countn++;
                        }
                        if (mboard2[i][j - 1] > 0) {
                            countp++;
                        }
                    }
                    if (valid_pos(i, j + 1)) {
                        if (mboard2[i][j + 1] < 0) {
                            countn++;
                        }
                        if (mboard2[i][j + 1] > 0) {
                            countp++;
                        }
                    }
                    if (countp > 0 && countn == 0) {
                        mboard[i][j] += countp;
                    } else if (countn > 0 && countp == 0) {
                        mboard[i][j] -= countn;
                    }
                }
            }
        }
    }
    // 21 Erosion
    for (int t = 0; t < 5; t++) {
        mboard2 = mboard;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (mboard2[i][j] > 0) {
                    int count = 0;
                    if (valid_pos(i - 1, j) && mboard2[i - 1][j] <= 0) {
                        count++;
                    }
                    if (valid_pos(i + 1, j) && mboard2[i + 1][j] <= 0) {
                        count++;
                    }
                    if (valid_pos(i, j - 1) && mboard2[i][j - 1] <= 0) {
                        count++;
                    }
                    if (valid_pos(i, j + 1) && mboard2[i][j + 1] <= 0) {
                        count++;
                    }
                    mboard[i][j] > count ? mboard[i][j] - count : 0;
                } else if (mboard2[i][j] < 0) {
                    int count = 0;
                    if (valid_pos(i - 1, j) && mboard2[i - 1][j] >= 0) {
                        count++;
                    }
                    if (valid_pos(i + 1, j) && mboard2[i + 1][j] >= 0) {
                        count++;
                    }
                    if (valid_pos(i, j - 1) && mboard2[i][j - 1] >= 0) {
                        count++;
                    }
                    if (valid_pos(i, j + 1) && mboard2[i][j + 1] >= 0) {
                        count++;
                    }
                    mboard[i][j] < -count ? mboard[i][j] + count : 0;
                }
            }
        }
    }
    int countb = 0, countw = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (mboard[i][j] > 0 && mboard[i][j] < 24)
                countb++;
            if (mboard[i][j] < 0 && mboard[i][j] > -24)
                countw++;
        }
    }
    
    return countb > countw + 5;
}

void channel_01(py::array_t<int> board, int x, int y, int turn) {
    auto b = board.mutable_unchecked<3>();
    set<pair<int, int>> live, died;

    function<bool(int, int, int)> checkDie = [&](int x, int y, int p) -> bool {
        int pp = 1 - p;
        if (live.count({x, y})) return false;
        if (died.count({x, y})) return true;
        died.insert({x, y});
        bool ans = true;
        vector<pair<int, int>> directions = {{x-1, y}, {x, y-1}, {x+1, y}, {x, y+1}};
        for (auto [dx, dy] : directions) {
            if (valid_pos(dx, dy)) {
                if (b(p, dx, dy) == 0 && b(pp, dx, dy) == 0) {
                    live.insert({x, y});
                    return false;
                }
                if (b(p, dx, dy) == 1) {
                    ans = ans & checkDie(dx, dy, p);
                }
            }
        }
        if (!ans) {
            died.erase({x, y});
            live.insert({x, y});
        }
        return ans;
    };

    function<void(int, int, int)> del_die = [&](int x, int y, int p) {
        b(p, x, y) = 0;
        b(3, x, y) = 0;
        vector<pair<int, int>> directions = {{x-1, y}, {x, y-1}, {x+1, y}, {x, y+1}};
        for (auto [dx, dy] : directions) {
            if (valid_pos(dx, dy) && b(p, dx, dy)) {
                del_die(dx, dy, p);
            }
        }
    };

    b(turn % 2, x, y) = 1;

    vector<pair<int, int>> directions = {{x-1, y}, {x, y-1}, {x+1, y}, {x, y+1}};
    for (auto [dx, dy] : directions) {
        if (valid_pos(dx, dy)) {
            if (turn % 2 == 1 && b(0, dx, dy) && checkDie(dx, dy, 0)) {
                del_die(dx, dy, 0);
            } else if (turn % 2 == 0 && b(1, dx, dy) && checkDie(dx, dy, 1)) {
                del_die(dx, dy, 1);
            }
        }
    }
}

void channel_3(py::array_t<int> board, int x, int y, int turn) {

    auto b = board.mutable_unchecked<3>();
    set<pair<int, int>> counted_empty;
    set<pair<int, int>> counted_pos;

    function<int(int, int, int)> check_liberty = [&](int x, int y, int p) -> int {
        int liberty = 0;
        int pp = (p == 0) ? 1 : 0;
        b(p, x, y) = 2;
        vector<pair<int, int>> directions = {{x-1, y}, {x, y-1}, {x+1, y}, {x, y+1}};
        
        for (auto [dx, dy] : directions) {
            if (valid_pos(dx, dy)) {
                if (b(pp, dx, dy) == 0 && b(p, dx, dy) == 0) {
                    if (counted_empty.find({dx, dy}) == counted_empty.end()) {
                        liberty += 1;
                        counted_empty.insert({dx, dy});
                    }
                } else if (b(p, dx, dy) == 1) {
                    liberty += check_liberty(dx, dy, p);
                }
            }
        }

        b(p, x, y) = 1;
        counted_pos.insert({x, y});
        return liberty;
    };

    function<void(int, int, int, int)> set_liberty = [&](int x, int y, int p, int liberty) {
        b(p, x, y) = 2;
        b(3, x, y) = min(6, liberty);
        vector<pair<int, int>> directions = {{x-1, y}, {x, y-1}, {x+1, y}, {x, y+1}};
        for (auto [dx, dy] : directions) {
            if (valid_pos(dx, dy) && b(p, dx, dy) == 1) {
                set_liberty(dx, dy, p, liberty);
            }
        }
        b(p, x, y) = 1;
    };

    if (b(0, x, y) == 0 && b(1, x, y) == 0) {
        return;
    }
    set_liberty(x, y, turn % 2, check_liberty(x, y, turn % 2));

    int pp = (turn % 2 == 0) ? 1 : 0;
    vector<pair<int, int>> directions = {{x-1, y}, {x, y-1}, {x+1, y}, {x, y+1}};
 
    for (auto [dx, dy] : directions) {
        counted_empty.clear();
        if (valid_pos(dx, dy) && b(pp, dx, dy) && counted_pos.find({dx, dy}) == counted_pos.end()) {
            set_liberty(dx, dy, pp, check_liberty(dx, dy, pp));
        }
    }
}

PYBIND11_MODULE(cpptools, m) {
    m.def("value_situation", &value_situation, "value_situation");
    m.def("channel_01", &channel_01, "channel_01");
    m.def("channel_3", &channel_3, "channel_3");
}
