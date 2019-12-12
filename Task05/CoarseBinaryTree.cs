using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;

namespace Task05
{
    internal class CoarseBstNode<TK, TV> where TK : IComparable<TK>
    {
        public TK Key;
        public TV Value;
        public CoarseBstNode<TK, TV> Parent;
        public CoarseBstNode<TK, TV> Left;
        public CoarseBstNode<TK, TV> Right;
        public CoarseBstNode(TK key, TV value)
        {
            Key = key;
            Value = value;
        }
        public CoarseBstNode(TK key, TV value, CoarseBstNode<TK, TV> parent)
        {
            Key = key;
            Value = value;
            Parent = parent;
        }
    }

    internal class CoarseBinaryTree<TK, TV> where TK : IComparable<TK>

    {
        internal CoarseBstNode<TK, TV> Root;
        private Mutex _mutex = new Mutex();

        internal CoarseBstNode<TK, TV> Find(TK key)
        {
            try
            {
                _mutex.WaitOne();
                var cur = Root;
                while (cur != null)
                {
                    var result = cur;
                    switch (cur.Key.CompareTo(key))
                    {
                        //Eq
                        case 0:
                            return result;
                        //Greater
                        case 1:
                            cur = cur.Left;
                            break;
                        // Less
                        case -1:
                            cur = cur.Right;
                            break;
                    }
                }

                Console.Write($"Element with key = {key} not found");
                return null;
            }
            finally
            {
                _mutex.ReleaseMutex();
            }
        }

        internal void Insert(TK key, TV value)
        {
            _mutex.WaitOne();
            try
            {
                CoarseBstNode<TK, TV> parent = null;
                var cur = Root;

                while (cur != null)
                {
                    parent = cur;

                    switch (cur.Key.CompareTo(key))
                    {
                        //Greater
                        case 1:
                            cur = cur.Left;
                            break;
                        // Less
                        case -1:
                            cur = cur.Right;
                            break;
                        //Eq
                        case 0:
                            cur.Value = value;
                            return;
                    }
                }

                if (parent == null)
                {
                    Root = new CoarseBstNode<TK, TV>(key, value);
                    return;
                }

                if (parent.Key.CompareTo(key) == 1)
                {
                    parent.Left = new CoarseBstNode<TK, TV>(key, value, parent);
                }
                else parent.Right = new CoarseBstNode<TK, TV>(key, value, parent);
            }
            finally
            {
                _mutex.ReleaseMutex();
            }
        }

        internal void Delete(TK key)
        {
            _mutex.WaitOne();
            try
            {
                var cur = Find(key);
                if (cur == null)
                {
                    return;
                }

                var parent = cur.Parent;

                if (cur.Left == null && cur.Right == null)
                {
                    if (parent == null)
                    {
                        Root = null;
                        return;
                    }

                    if (cur == parent.Left)
                    {
                        parent.Left = null;
                    }

                    if (cur == parent.Right)
                    {
                        parent.Right = null;
                    }
                }
                else if (cur.Left == null || cur.Right == null)
                {
                    if (cur.Left == null)
                    {
                        if (parent == null)
                        {
                            Root = cur.Right;
                            if (cur.Right != null) cur.Right.Parent = null;
                            return;
                        }

                        if (cur == parent.Left)
                        {
                            parent.Left = cur.Right;
                        }
                        else
                        {
                            parent.Right = cur.Right;
                        }

                        if (cur.Right != null) cur.Right.Parent = parent;
                    }
                    else
                    {
                        if (parent == null)
                        {
                            Root = cur.Left;
                            cur.Left.Parent = null;
                            return;
                        }

                        if (cur == parent.Left)
                        {
                            parent.Left = cur.Left;
                        }
                        else
                        {
                            parent.Right = cur.Left;
                        }

                        cur.Left.Parent = parent;
                    }
                }
                else
                {
                    var successor = Min(cur.Right);
                    cur.Key = successor.Key;
                    cur.Value = successor.Value;
                    if (successor == successor.Parent.Left)
                    {
                        successor.Parent.Left = successor.Right;
                        if (successor.Right != null)
                        {
                            successor.Right.Parent = successor.Parent;
                        }
                    }
                    else
                    {
                        successor.Parent.Right = successor.Right;
                        if (successor.Right != null)
                            successor.Right.Parent = successor.Parent;
                    }
                }
            }
            finally
            {
                _mutex.ReleaseMutex();
            }
            
        }

        private static CoarseBstNode<TK, TV> Min(CoarseBstNode<TK, TV> cur)
        {
            while (true)
            {
                if (cur.Left == null) return cur;
                cur = cur.Left;
            }
        }
    }
}