    #include <iostream>
    #include <unordered_map>
    #include <set>
    #include <string>
    #include <cmath>
    #include <fstream>
    #include <sstream>
    #include <regex>
    #include <vector>
    #include <algorithm>

    using namespace std;

    // Function to calculate cosine similarity
    double cosineSimilarity(const vector<double>& a, const vector<double>& b) {
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return (normA && normB) ? dot / (sqrt(normA) * sqrt(normB)) : 0.0;
    }

    // Enhanced Neural Network Chatbot
    class NeuralNetChatbot {
    private:
        unordered_map<string, int> wordToIndex;
        unordered_map<string, unordered_map<string, int>> bigramFreq;
        unordered_map<string, unordered_map<string, int>> trigramFreq;
        vector<vector<double>> weights;
        unordered_map<string, string> responseMap;
        vector<string> responses;
        set<string> loadedPairs;
        vector<pair<string, string>> newTrainingData;
        double learningRate = 0.1;
        const int MAX_VOCAB_SIZE = 10000;
        const double CONFIDENCE_THRESHOLD = 0.65;
        const int RECURSION_LIMIT = 5;

    public:
        NeuralNetChatbot() {
            responses.push_back("I'm not sure about that. Can you explain more?");
            cout << "Initializing chatbot..." << endl;
            loadTrainingData();
            cout << "Initialization complete." << endl;
        }

        void train(string userInput, string response) {
            responseMap[userInput] = response;
            int responseIndex = findOrAddResponse(response);
            vector<string> words = tokenize(userInput);
            for (string& word : words) {
                int index = getWordIndex(word);
                if (index != -1) {
                    ensureWeightSize(index, responseIndex);
                    weights[index][responseIndex] += learningRate;
                }
            }
            updateNgramFrequencies(words);
            string pairKey = userInput + "||" + response;
            if (loadedPairs.find(pairKey) == loadedPairs.end()) {
                newTrainingData.push_back({userInput, response});
            }
            saveTrainingData();
        }

        string predict(string userInput) {
            vector<string> tokens = tokenize(userInput);
            string response = processSubstrings(tokens, 0);
            return response.empty() ? "I'm not sure about that. Can you explain more?" : response;
        }

        string processSubstrings(vector<string> tokens, int depth) {
            if (tokens.empty() || depth > RECURSION_LIMIT) return "";
            string bestMatch = findBestSubstringMatch(tokens);

            if (!bestMatch.empty() && responseMap.find(bestMatch) != responseMap.end()) {
                vector<string> remainingTokens = removeSubstring(tokens, bestMatch);
                string remainingResponse = processSubstrings(remainingTokens, depth + 1);
                return responseMap[bestMatch] + (remainingResponse.empty() ? "" : " " + remainingResponse);
            } else {
                return handleUnmatchedTokens(tokens);
            }
        }

        string handleUnmatchedTokens(const vector<string>& tokens) {
            vector<double> scores(responses.size(), 0.0);
            for (const string& word : tokens) {
                if (wordToIndex.find(word) != wordToIndex.end()) {
                    int index = wordToIndex[word];
                    for (size_t i = 0; i < responses.size(); i++) {
                        scores[i] += weights[index][i];
                    }
                }
            }
            for (double& score : scores) score = sigmoid(score);
            double maxScore = 0.0;
            string bestResponse = "";
            for (size_t i = 0; i < scores.size(); i++) {
                if (scores[i] > maxScore && scores[i] >= CONFIDENCE_THRESHOLD) {
                    maxScore = scores[i];
                    bestResponse = responses[i];
                }
            }
            return bestResponse;
        }

        vector<string> removeSubstring(const vector<string>& tokens, const string& substring) {
            vector<string> remaining;
            vector<string> subTokens = tokenize(substring);
            size_t i = 0;
            while (i < tokens.size()) {
                bool match = true;
                for (size_t j = 0; j < subTokens.size() && (i + j) < tokens.size(); j++) {
                    if (tokens[i + j] != subTokens[j]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    i += subTokens.size();
                } else {
                    remaining.push_back(tokens[i]);
                    i++;
                }
            }
            return remaining;
        }

        string findBestSubstringMatch(const vector<string>& tokens) {
            string bestMatch = "";
            size_t maxLength = 0;
            for (size_t len = tokens.size(); len > 0; len--) {
                for (size_t i = 0; i <= tokens.size() - len; i++) {
                    string substring = joinTokens(tokens, i, len);
                    if (responseMap.find(substring) != responseMap.end()) {
                        if (len > maxLength) {
                            maxLength = len;
                            bestMatch = substring;
                        }
                    }
                }
            }
            return bestMatch;
        }

        string joinTokens(const vector<string>& tokens, size_t start, size_t length) {
            string result = "";
            for (size_t i = start; i < start + length; i++) {
                result += tokens[i];
                if (i < start + length - 1) result += " ";
            }
            return result;
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
                    responseMap[input] = response;
                    int responseIndex = findOrAddResponse(response);
                    vector<string> words = tokenize(input);
                    for (string& word : words) {
                        int index = getWordIndex(word);
                        ensureWeightSize(index, responseIndex);
                        weights[index][responseIndex] += learningRate;
                    }
                    updateNgramFrequencies(words);
                }
            }
            file.close();
        }

        vector<string> tokenize(string input) {
            vector<string> tokens;
            stringstream ss(input);
            string token;
            while (ss >> token) {
                tokens.push_back(token);
            }
            return tokens;
        }

        int findOrAddResponse(string response) {
            for (size_t i = 0; i < responses.size(); i++) {
                if (responses[i] == response) return i;
            }
            responses.push_back(response);
            return responses.size() - 1;
        }

        int getWordIndex(string word) {
            if (wordToIndex.find(word) == wordToIndex.end()) {
                if (wordToIndex.size() < MAX_VOCAB_SIZE) {
                    int index = wordToIndex.size();
                    wordToIndex[word] = index;
                    weights.resize(wordToIndex.size(), vector<double>(responses.size(), 0.0));
                } else {
                    return -1;
                }
            }
            return wordToIndex[word];
        }

        void ensureWeightSize(int wordIndex, int responseIndex) {
            if (weights[wordIndex].size() <= responseIndex) {
                weights[wordIndex].resize(responseIndex + 1, 0.0);
            }
        }

        void updateNgramFrequencies(const vector<string>& words) {
            for (size_t i = 0; i < words.size() - 1; i++) {
                bigramFreq[words[i]][words[i + 1]]++;
                if (i < words.size() - 2) {
                    trigramFreq[words[i] + " " + words[i + 1]][words[i + 2]]++;
                }
            }
        }

        double sigmoid(double x) {
            return 1.0 / (1.0 + exp(-x));
        }

        void saveTrainingData() {
            ofstream file("mastertrain.txt", ios::app);
            for (const auto& pair : newTrainingData) {
                file << pair.first << "||" << pair.second << "\n";
            }
            newTrainingData.clear();
            file.close();
        }
    };

    int main() {
        NeuralNetChatbot chatbot;
        string input;
        cout << "Chatbot initialized. Type 'exit' to quit." << endl;
        while (true) {
            cout << "You: ";
            getline(cin, input);
            if (input == "exit") break;
            string response = chatbot.predict(input);
            cout << "Bot: " << response << endl;
        }
        return 0;
    }
