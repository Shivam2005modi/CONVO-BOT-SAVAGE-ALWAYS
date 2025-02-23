// Updated Chatbot with Dynamic Similarity Check
#include <bits/stdc++.h>
#include <csignal>
using namespace std;

// Activation Function: Sigmoid
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Stopwords Set
unordered_set<string> stopwords = {"the", "is", "and", "a", "an", "of", "in"};

// Tokenization Function
vector<string> tokenize(string text) {
    vector<string> tokens;
    stringstream ss(text);
    string word;
    while (ss >> word) {
        transform(word.begin(), word.end(), word.begin(), ::tolower);
        if (stopwords.find(word) == stopwords.end()) {
            tokens.push_back(word);
        }
    }
    return tokens;
}

// Levenshtein Distance Function for Character-Level Similarity
int levenshteinDistance(const string& a, const string& b) {
    int m = a.size(), n = b.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
            }
        }
    }
    return dp[m][n];
}

// Character-Level Similarity Function
double characterSimilarity(const string& a, const string& b) {
    int distance = levenshteinDistance(a, b);
    int maxLen = max(a.length(), b.length());
    return maxLen == 0 ? 1.0 : (1.0 - (double)distance / maxLen);
}

// Neural Network Chatbot
class NeuralNetChatbot {
private:
    unordered_map<string, int> wordToIndex;
    unordered_map<string, unordered_map<string, int>> bigramFreq;
    unordered_map<string, unordered_map<string, int>> trigramFreq;
    unordered_map<string, string> inputToResponse;
    vector<vector<double>> weights;
    set<string> loadedPairs;
    vector<pair<string, string>> newTrainingData;
    double learningRate = 0.1;
    const int MAX_VOCAB_SIZE = 10000;
    const double CONFIDENCE_THRESHOLD = 0.65;

public:
    NeuralNetChatbot() {
        loadTrainingData();
    }

    int getWordIndex(string word) {
        if (wordToIndex.find(word) == wordToIndex.end()) {
            if (wordToIndex.size() >= MAX_VOCAB_SIZE) return -1;
            int newIndex = wordToIndex.size();
            wordToIndex[word] = newIndex;
            for (auto& row : weights) row.push_back(0.0);
            weights.push_back(vector<double>(1, 0.0));
        }
        return wordToIndex[word];
    }

    void train(string userInput, string response) {
        inputToResponse[userInput] = response;
        vector<string> words = tokenize(userInput);
        for (string& word : words) {
            int index = getWordIndex(word);
            if (index != -1) weights[index][0] += learningRate;
        }
        updateNgramFrequencies(words);
        string pairKey = userInput + "||" + response;
        if (loadedPairs.find(pairKey) == loadedPairs.end()) {
            newTrainingData.push_back({userInput, response});
        }
    }

    void updateNgramFrequencies(vector<string>& words) {
        for (size_t i = 0; i + 1 < words.size(); i++) {
            bigramFreq[words[i]][words[i + 1]]++;
            if (i + 2 < words.size()) {
                trigramFreq[words[i] + " " + words[i + 1]][words[i + 2]]++;
            }
        }
    }

    string predict(string userInput) {
        double maxSimilarity = 0.0;
        string bestMatch = "";

        for (const auto& pair : inputToResponse) {
            double similarity = characterSimilarity(userInput, pair.first);
            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                bestMatch = pair.first;
            }
        }

        if (maxSimilarity >= CONFIDENCE_THRESHOLD) {
            return inputToResponse[bestMatch];
        } else {
            return "I'm not sure about that. Can you explain more?";
        }
    }

    void saveTrainingData() {
        ofstream file("mastertrain.txt", ios::app);
        if (file.is_open()) {
            for (auto& pair : newTrainingData) {
                string pairKey = pair.first + "||" + pair.second;
                if (loadedPairs.find(pairKey) == loadedPairs.end()) {
                    file << pair.first << "||" << pair.second << endl;
                    loadedPairs.insert(pairKey);
                }
            }
            newTrainingData.clear();
            file.close();
        }
    }

    void loadTrainingData() {
        ifstream file("mastertrain.txt");
        string line;
        while (getline(file, line)) {
            size_t delimiter = line.find("||");
            if (delimiter != string::npos) {
                string input = line.substr(0, delimiter);
                string response = line.substr(delimiter + 2);
                string pairKey = input + "||" + response;
                loadedPairs.insert(pairKey);
                train(input, response);
            }
        }
        file.close();
    }
};

NeuralNetChatbot* globalBot = nullptr;

void handleExit(int signal) {
    if (globalBot) {
        globalBot->saveTrainingData();
        cout << "\nChatbot: Auto-saved before exit. Goodbye!" << endl;
    }
    exit(signal);
}

int main() {
    NeuralNetChatbot bot;
    globalBot = &bot;
    signal(SIGINT, handleExit);

    string userInput;
    cout << "Chatbot: Hello! Let's chat. Type 'exit' to stop." << endl;
    while (true) {
        cout << "You: ";
        getline(cin, userInput);
        if (userInput == "exit") break;
        string botResponse = bot.predict(userInput);
        cout << "Chatbot: " << botResponse << endl;
        if (botResponse == "I'm not sure about that. Can you explain more?") {
            cout << "Teach me! What should I reply?\nYou: ";
            string response;
            getline(cin, response);
            bot.train(userInput, response);
            cout << "Chatbot: Got it! I'll remember that." << endl;
        }
    }
    bot.saveTrainingData();
    cout << "Chatbot: It was great talking to you. Goodbye!" << endl;
    return 0;
}
