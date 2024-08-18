#include<iostream>
#include<unordered_map>
using namespace std;


int main() {
    unordered_map<int,int> mp;
    // if(mp[10] == 1) cout<<10<<endl;
    if(mp.find(10) == mp.end()) cout<<'h'<<endl;
    cout<<mp[10]<<endl;
    return 0;
}