﻿using System;
using System.Collections.Generic;
 using System.Threading;
 using System.Threading.Tasks;

namespace Task05
{
    //Helper methods for generic binary search tree testing
    public class TestHelper<T>
        where T : IBinarySearchTree<int, int>, new()
    {
        private readonly Random _rnd = new Random();

        private List<int> GenerateListOfKeys(int numKeys)
        {
            var testKeys = new List<int>();
            for (int i = 0; i < numKeys; i++)
            {
                var key = _rnd.Next(numKeys);
                testKeys.Add(key);
            }
            return testKeys;
        }

        public bool FoundAllInserted(int numKeys)
        {
            var testTree = new T();
            var testKeys = GenerateListOfKeys(numKeys);
            var foundAll = true;

            //Creating insertion tasks
            var treeInsertionTasks = new List<Task>();
            foreach (var key in testKeys)
            {
                treeInsertionTasks.Add(Task.Run(() => {
                    testTree.Insert(key, 0);
                }));
            }
            Task.WaitAll(treeInsertionTasks.ToArray());
            
            //Creating search tasks
            var treeFindTasks = new List<Task>();
            foreach (var key in testKeys)
            {
                treeFindTasks.Add(Task.Run(() => {
                    if (testTree.Find(key) == null)
                    {
                        foundAll = false;
                    }
                }));
            }
            Task.WaitAll(treeFindTasks.ToArray());
            
            return foundAll;
        }

        public bool IsEmptyAfterConcurrentInsertionRemoval(int numKeys, int numTasks)
        {
            var testTree = new T();
            var treeInsertionAndRemovalTasks = new List<Task>();

            //Creating removal and insertion tasks
            for (int i = 0; i < numTasks; i++)
            {
                treeInsertionAndRemovalTasks.Add(Task.Run(() =>
                {
                    var testKeys = GenerateListOfKeys(numKeys);
                    foreach (var key in testKeys)
                    {
                        testTree.Insert(key, 0);
                    }
                    foreach (var key in testKeys)
                    {
                        testTree.Remove(key);
                    }
                }));
            }
            Task.WaitAll(treeInsertionAndRemovalTasks.ToArray());

            return testTree.IsEmpty();
        }

        public bool IsEmptyAfterInsertionRemoval(int numKeys)
        {
            var testTree = new T();
            var testKeys = GenerateListOfKeys(numKeys);

            //Creating insertion tasks
            var treeInsertionTasks = new List<Task>();
            foreach (var key in testKeys)
            {
                treeInsertionTasks.Add(Task.Run(() => {
                    testTree.Insert(key, 0);
                }));
            }
            Task.WaitAll(treeInsertionTasks.ToArray());
            
            //Creating removal tasks
            var treeRemovalTasks = new List<Task>();
            foreach (var key in testKeys)
            {
                treeRemovalTasks.Add(Task.Run(() => {
                    testTree.Remove(key);
                }));
            }
            Task.WaitAll(treeRemovalTasks.ToArray());
            
            return testTree.IsEmpty();
        }
    }
}
